"""Tests for inotify_ping: pure logic (parse/classify/config/watch_target) + real-fd integration."""

from __future__ import annotations

import sys
import pytest

import asyncio
import struct

import pytest

from inotify_ping import (
    IN_CLOSE_WRITE,
    IN_DELETE,
    IN_MODIFY,
    IN_MOVED_TO,
    InotifyPingPlugin,
    classify,
    has_terminal_marker,
    load_config,
    parse_events,
    watch_target,
)
from corvidae.channel import ChannelRegistry

# --- fakes -------------------------------------------------------------------


class FakeChannel:
    def __init__(self, cid: str):
        self.id = cid


class FakeAHook:
    def __init__(self, channel: FakeChannel, calls: list, resolvable: bool = True):
        self._channel = channel
        self._calls = calls
        self._resolvable = resolvable

    async def resolve_channel(self, channel_id: str):
        if not self._resolvable:
            return None
        return self._channel if channel_id == self._channel.id else None

    async def on_notify(self, channel, source, text, tool_call_id, annotations):
        self._calls.append(
            {"channel": channel, "source": source, "text": text, "tool_call_id": tool_call_id, "annotations": annotations}
        )


class FakeAgent:
    def __init__(self):
        self.started = True
        self._started_event = asyncio.Event()
        self._started_event.set()


class FakePM:
    def __init__(self, channel_id: str, resolvable: bool = True):
        self.calls: list = []
        self._channel = FakeChannel(channel_id)
        self.ahook = FakeAHook(self._channel, self.calls, resolvable=resolvable)
        self._agent = FakeAgent()
        self._registry = ChannelRegistry()

    def get_plugin(self, name: str):
        if name == "agent":
            return self._agent
        if name == "registry":
            return self._registry
        return None


CFG = {"enabled": True, "channel": "test:chan", "debounce_seconds": 0.2, "paths": None}


def make_plugin(tmp_path, pm=None, **overrides):
    pm = pm or FakePM("test:chan")
    f = tmp_path / "job.log"
    f.write_text("start\n")
    cfg = dict(CFG, paths=[str(f)])
    cfg.update(overrides)
    plugin = InotifyPingPlugin()
    return plugin, pm, f, cfg


async def _wait_armed(plugin, timeout=2.0):
    """Wait until the plugin's inotify fd is armed (background task completed)."""
    deadline = asyncio.get_event_loop().time() + timeout
    while plugin._fd is None:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError("plugin did not arm in time")
        await asyncio.sleep(0.01)


# --- pure: inotify event parsing --------------------------------------------


def test_parse_events_single_no_name():
    buf = struct.pack("iIII", 3, IN_MODIFY, 0, 0)
    assert parse_events(buf) == [(3, IN_MODIFY, "")]


def test_parse_events_with_name():
    # the name is a null-terminated C string; the trailing NUL is stripped
    buf = struct.pack("iIII", 5, IN_CLOSE_WRITE, 0, 4) + b"abc\x00"
    assert parse_events(buf) == [(5, IN_CLOSE_WRITE, "abc")]


def test_parse_events_multiple():
    e1 = struct.pack("iIII", 1, IN_MODIFY, 0, 0)
    e2 = struct.pack("iIII", 2, IN_MOVED_TO, 0, 0)
    assert parse_events(e1 + e2) == [(1, IN_MODIFY, ""), (2, IN_MOVED_TO, "")]


def test_parse_events_empty():
    assert parse_events(b"") == []


# --- pure: event classification ---------------------------------------------


def test_classify_write_masks():
    assert classify(IN_MODIFY) == "write"
    assert classify(IN_CLOSE_WRITE) == "write"
    assert classify(IN_MOVED_TO) == "write"
    assert classify(IN_MODIFY | IN_CLOSE_WRITE) == "write"


def test_classify_non_write():
    assert classify(IN_DELETE) is None
    assert classify(0) is None


# --- pure: watch target (file vs directory watch) ---------------------------


def test_watch_target_existing_file(tmp_path):
    f = tmp_path / "job.log"
    f.write_text("x")
    assert watch_target(str(f)) == (str(f), None)


def test_watch_target_missing_file_existing_dir(tmp_path):
    # the file is absent but its parent dir exists -> directory watch on the basename
    missing = tmp_path / "job.log"
    assert watch_target(str(missing)) == (str(tmp_path), "job.log")


def test_watch_target_missing_file_missing_dir(tmp_path):
    # neither file nor parent dir exists -> return the path unchanged (the
    # caller surfaces the ENOENT)
    missing = tmp_path / "nodir" / "job.log"
    assert watch_target(str(missing)) == (str(missing), None)


# --- pure: config -----------------------------------------------------------


def test_load_config_disabled_is_inert():
    cfg = load_config({"enabled": False})
    assert cfg.enabled is False
    assert cfg.channel == ""
    assert cfg.paths == []


def test_load_config_valid():
    cfg = load_config({"enabled": True, "channel": "c", "debounce_seconds": 3, "paths": ["/a", "/b"]})
    assert cfg.enabled is True
    assert cfg.channel == "c"
    assert cfg.debounce_seconds == 3.0
    assert cfg.paths == ["/a", "/b"]


def test_load_config_defaults_off():
    assert load_config({}).enabled is False


def test_load_config_bad_enabled():
    with pytest.raises(ValueError):
        load_config({"enabled": "yes"})


def test_load_config_bad_channel():
    with pytest.raises(ValueError):
        load_config({"enabled": True, "channel": "", "paths": ["/a"]})


def test_load_config_bad_debounce():
    with pytest.raises(ValueError):
        load_config({"enabled": True, "channel": "c", "debounce_seconds": -1, "paths": ["/a"]})


def test_load_config_bool_debounce_rejected():
    with pytest.raises(ValueError):
        load_config({"enabled": True, "channel": "c", "debounce_seconds": True, "paths": ["/a"]})


def test_load_config_bad_paths():
    with pytest.raises(ValueError):
        load_config({"enabled": True, "channel": "c", "paths": []})


# --- disabled plugin is inert ------------------------------------------------


async def test_disabled_is_inert(tmp_path):
    plugin = InotifyPingPlugin()
    pm = FakePM("test:chan")
    await plugin.on_init(pm, {"inotify_ping": {"enabled": False}})
    await plugin.on_start({})
    assert plugin._fd is None
    assert plugin._thread is None


# --- integration: real inotify fd -------------------------------------------


async def test_settle_pings_once(tmp_path):
    plugin, pm, f, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    f.write_text("start\nwork\n")
    await asyncio.sleep(0.7)
    await plugin.on_stop()
    assert len(pm.calls) == 1
    c = pm.calls[0]
    assert c["source"] == "inotify"
    assert c["annotations"] == {"origin": "inotify", "path": str(f), "event": "settled"}
    assert str(f) in c["text"]


async def test_burst_coalesces_to_one_ping(tmp_path):
    plugin, pm, f, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    for i in range(3):
        with open(f, "a") as fh:
            fh.write(f"line{i}\n")
    await asyncio.sleep(0.7)
    await plugin.on_stop()
    assert len(pm.calls) == 1


async def test_rearm_after_settle(tmp_path):
    plugin, pm, f, cfg = make_plugin(tmp_path, debounce_seconds=0.15)
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    f.write_text("a\n")
    await asyncio.sleep(0.6)
    assert len(pm.calls) == 1
    f.write_text("b\n")
    await asyncio.sleep(0.6)
    assert len(pm.calls) == 2
    await plugin.on_stop()


async def test_channel_unresolvable_drops_ping(tmp_path):
    # A no-colon id is not transport:scope, so resolve_persisted_channel
    # returns None (no registry entry, no transport, no get_or_create).
    pm = FakePM("test:chan", resolvable=False)
    plugin, pm, f, cfg = make_plugin(tmp_path, pm=pm, channel="no_colon")
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    f.write_text("work\n")
    await asyncio.sleep(0.7)
    await plugin.on_stop()
    assert len(pm.calls) == 0


# --- integration: directory watch (missing file at boot) --------------------


async def test_directory_watch_file_created_after_arm(tmp_path):
    """A missing file at boot degrades to a directory watch; the file can
    appear and settle after arming (graceful degradation, not a silent 0/N)."""
    pm = FakePM("test:chan")
    target = tmp_path / "late.log"  # does NOT exist at arm time
    cfg = {"enabled": True, "channel": "test:chan", "debounce_seconds": 0.2, "paths": [str(target)]}
    plugin = InotifyPingPlugin()
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    # the watch degraded to a directory watch filtered on the basename
    assert any(fn == "late.log" for entries in plugin._wd_info.values() for _p, fn in entries)
    # now the job creates + writes the file
    target.write_text("start\nwork\n")
    await asyncio.sleep(0.7)
    await plugin.on_stop()
    assert len(pm.calls) == 1
    assert str(target) in pm.calls[0]["text"]
    assert pm.calls[0]["annotations"]["path"] == str(target)


async def test_directory_watch_ignores_other_files(tmp_path):
    """A directory watch filters for the target basename; writes to OTHER files
    in the same directory do not trigger a ping."""
    pm = FakePM("test:chan")
    target = tmp_path / "target.log"  # does NOT exist at arm time
    other = tmp_path / "other.log"
    other.write_text("x\n")  # exists
    cfg = {"enabled": True, "channel": "test:chan", "debounce_seconds": 0.2, "paths": [str(target)]}
    plugin = InotifyPingPlugin()
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    # write to the OTHER file — must NOT trigger a ping
    other.write_text("y\n")
    await asyncio.sleep(0.7)
    assert len(pm.calls) == 0, "ping fired for a non-target file"
    # now write to the target — SHOULD trigger a ping
    target.write_text("work\n")
    await asyncio.sleep(0.7)
    await plugin.on_stop()
    assert len(pm.calls) == 1
    assert str(target) in pm.calls[0]["text"]


# --- pure: terminal marker ---------------------------------------------------


def test_has_terminal_marker_present(tmp_path):
    f = tmp_path / "job.log"
    f.write_text("start\nwork\ndone\n")
    assert has_terminal_marker(str(f), "done") is True


def test_has_terminal_marker_absent(tmp_path):
    f = tmp_path / "job.log"
    f.write_text("start\nwork\n")
    assert has_terminal_marker(str(f), "done") is False


def test_has_terminal_marker_missing_file(tmp_path):
    missing = tmp_path / "nope.log"
    assert has_terminal_marker(str(missing), "done") is False


def test_has_terminal_marker_ignores_trailing_blank(tmp_path):
    f = tmp_path / "job.log"
    f.write_text("start\nwork\ndone\n\n\n")
    assert has_terminal_marker(str(f), "done") is True


def test_has_terminal_marker_substring(tmp_path):
    f = tmp_path / "job.log"
    f.write_text("start\n[done] all 5 folds\n")
    assert has_terminal_marker(str(f), "done") is True


def test_load_config_terminal_marker():
    cfg = load_config({"enabled": True, "channel": "c", "paths": ["/a"], "terminal_marker": "done"})
    assert cfg.terminal_marker == "done"


def test_load_config_no_terminal_marker_defaults_none():
    cfg = load_config({"enabled": True, "channel": "c", "paths": ["/a"]})
    assert cfg.terminal_marker is None


def test_load_config_bad_terminal_marker():
    with pytest.raises(ValueError):
        load_config({"enabled": True, "channel": "c", "paths": ["/a"], "terminal_marker": ""})


# --- integration: terminal marker criterion ----------------------------------


async def test_settle_with_marker_pings_on_marker(tmp_path):
    """With a terminal marker configured, the settle pings once the marker
    line is present (deterministic completion, not just a quiet period)."""
    plugin, pm, f, cfg = make_plugin(tmp_path, terminal_marker="done")
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    f.write_text("start\nwork\ndone\n")
    await asyncio.sleep(0.7)
    await plugin.on_stop()
    assert len(pm.calls) == 1


async def test_settle_with_marker_no_ping_without_marker(tmp_path):
    """With a terminal marker configured, a quiet file that LACKS the marker
    does NOT ping (a mid-run pause is not a completion)."""
    plugin, pm, f, cfg = make_plugin(tmp_path, terminal_marker="done")
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    f.write_text("start\nwork\n")  # no "done"
    await asyncio.sleep(0.7)
    await plugin.on_stop()
    assert len(pm.calls) == 0


async def test_settle_no_marker_uses_debounce_only(tmp_path):
    """Without a terminal marker, the settle pings on the quiet period alone
    (the original behavior is preserved)."""
    plugin, pm, f, cfg = make_plugin(tmp_path)  # no terminal_marker
    await plugin.on_init(pm, {"inotify_ping": cfg})
    await plugin.on_start({})
    await _wait_armed(plugin)
    f.write_text("start\nwork\n")
    await asyncio.sleep(0.7)
    await plugin.on_stop()
    assert len(pm.calls) == 1


# --- startup deadlock regression ---------------------------------------------


class UnstartedFakeAgent:
    """Agent that has not yet started — the real Runtime ordering."""
    def __init__(self):
        self.started = False
        self._started_event = asyncio.Event()
        # event is NOT set — the agent hasn't started yet


class UnstartedFakePM:
    """PM whose agent is unstarted, matching Runtime.start ordering."""
    def __init__(self, channel_id: str):
        self.calls: list = []
        self._channel = FakeChannel(channel_id)
        self.ahook = FakeAHook(self._channel, self.calls)
        self._agent = UnstartedFakeAgent()

    def get_plugin(self, name: str):
        return self._agent if name == "agent" else None


async def test_on_start_returns_before_agent_starts(tmp_path):
    """on_start must not block on agent._started_event (deadlock regression).

    Runtime.start awaits the on_start broadcast, then calls agent.on_start()
    which sets _started_event. If inotify_ping.on_start awaits that event,
    the agent can never start. on_start must return promptly and arm the
    watches in a background task.
    """
    pm = UnstartedFakePM("test:chan")
    f = tmp_path / "job.log"
    f.write_text("start\n")
    cfg = {"enabled": True, "channel": "test:chan", "debounce_seconds": 0.2, "paths": [str(f)]}
    plugin = InotifyPingPlugin()
    await plugin.on_init(pm, {"inotify_ping": cfg})

    # on_start must return promptly even though the agent is unstarted
    await asyncio.wait_for(plugin.on_start({}), timeout=1.0)

    # Watches should NOT be armed yet (agent hasn't started)
    assert plugin._fd is None, "watches armed before agent started"

    # Now simulate the agent starting
    pm._agent.started = True
    pm._agent._started_event.set()

    # Give the background task time to arm the watches
    await asyncio.sleep(0.1)
    assert plugin._fd is not None, "watches not armed after agent started"

    await plugin.on_stop()


async def test_on_stop_cancels_pending_arm(tmp_path):
    """on_stop must cancel the background arm task if the agent never starts."""
    pm = UnstartedFakePM("test:chan")
    f = tmp_path / "job.log"
    f.write_text("start\n")
    cfg = {"enabled": True, "channel": "test:chan", "debounce_seconds": 0.2, "paths": [str(f)]}
    plugin = InotifyPingPlugin()
    await plugin.on_init(pm, {"inotify_ping": cfg})

    await asyncio.wait_for(plugin.on_start({}), timeout=1.0)
    # Agent never starts — on_stop should clean up the pending task
    await plugin.on_stop()
    assert plugin._fd is None
