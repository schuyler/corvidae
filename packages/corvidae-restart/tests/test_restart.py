"""Tests for the restart plugin: loop guard, pre-flight, marker, notification."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from corvidae_restart import (
    RestartPlugin,
    _is_number,
    read_boot_status,
    read_jsonl,
    truncate_stderr,
)


# --- fakes -------------------------------------------------------------------


class FakeChannel:
    def __init__(self, cid: str):
        self.id = cid


class FakeRegistry:
    def __init__(self, channels: dict[str, FakeChannel] | None = None):
        self._channels = channels or {}

    def get(self, cid: str):
        return self._channels.get(cid)


class FakeAgent:
    def __init__(self):
        self.tools = {}


class FakeAHook:
    def __init__(self):
        self.sends: list[dict] = []
        self.notifies: list[dict] = []

    async def send(self, channel, event):
        self.sends.append({"channel": channel, "event": event})

    async def on_notify(self, channel, source, text, tool_call_id, meta):
        self.notifies.append(
            {
                "channel": channel,
                "source": source,
                "text": text,
                "tool_call_id": tool_call_id,
                "meta": meta,
            }
        )


class FakePM:
    def __init__(self, channels: dict[str, FakeChannel] | None = None):
        self.ahook = FakeAHook()
        self._registry = FakeRegistry(channels)
        self._agent = FakeAgent()
        self._plugins = {
            "registry": self._registry,
            "agent": self._agent,
        }

    def get_plugin(self, name: str):
        return self._plugins.get(name)

    def list_name_plugin(self):
        return self._plugins.items()


CFG = {
    "_base_dir": "/tmp/test-restart",
    "restart": {
        "home_channel": "test:home",
        "config_path": "agent.yaml",
        "min_uptime_seconds": 120,
        "max_restarts": 3,
        "window_seconds": 21600,
        "preflight_timeout_seconds": 120,
        "notify_delay_seconds": 0,
    },
    "plugins": {"disabled": []},
}


def make_plugin(tmp_path, pm=None, **overrides):
    pm = pm or FakePM({"test:home": FakeChannel("test:home")})
    cfg = dict(CFG)
    cfg["_base_dir"] = str(tmp_path)
    if overrides:
        cfg["restart"] = {**CFG["restart"], **overrides}
    plugin = RestartPlugin()
    return plugin, pm, cfg


# --- pure: helpers -----------------------------------------------------------


def test_truncate_stderr_short():
    assert truncate_stderr("hello") == "hello"


def test_truncate_stderr_long():
    text = "x" * 5000
    result = truncate_stderr(text)
    # 400 head + elision marker + 1599 tail = ~2000 + marker overhead
    assert len(result) <= 2100
    assert result.startswith("x" * 400)
    assert "truncated" in result
    assert result.endswith("x" * 1599)


def test_is_number():
    assert _is_number(1)
    assert _is_number(1.5)
    assert not _is_number("1")
    assert not _is_number(None)
    assert not _is_number(True)
    assert not _is_number(False)


def test_read_jsonl_missing(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_read_jsonl_valid(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"a": 1}\n{"b": 2}\n\nnot json\n')
    records = read_jsonl(f)
    assert records == [{"a": 1}, {"b": 2}]


def test_read_boot_status_empty(tmp_path):
    mode, abnormal = read_boot_status(tmp_path / "nope.jsonl")
    assert mode is None
    assert abnormal is False


def test_read_boot_status_normal(tmp_path):
    f = tmp_path / "boot.jsonl"
    f.write_text(
        json.dumps({"mode": "normal", "ts": 1}) + "\n"
        + json.dumps({"service_result": "success", "ts": 2}) + "\n"
        + json.dumps({"mode": "normal", "ts": 3}) + "\n"
    )
    mode, abnormal = read_boot_status(f)
    assert mode == "normal"
    assert abnormal is False


def test_read_boot_status_abnormal_previous(tmp_path):
    f = tmp_path / "boot.jsonl"
    f.write_text(
        json.dumps({"mode": "normal", "ts": 1}) + "\n"
        + json.dumps({"service_result": "failure", "ts": 2}) + "\n"
        + json.dumps({"mode": "normal", "ts": 3}) + "\n"
    )
    mode, abnormal = read_boot_status(f)
    assert mode == "normal"
    assert abnormal is True


# --- loop guard ----------------------------------------------------------------


async def test_loop_guard_no_human_message(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._booted_mono = 0
    plugin._monotonic = lambda: 200  # uptime 200s > min 120s
    # No human message
    refusal = plugin._loop_guard_refusal()
    assert refusal is not None
    assert "no human message" in refusal


async def test_loop_guard_min_uptime(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._user_message_seen = True
    plugin._booted_mono = 0
    plugin._monotonic = lambda: 50  # uptime 50s < min 120s
    refusal = plugin._loop_guard_refusal()
    assert refusal is not None
    assert "uptime" in refusal


async def test_loop_guard_max_restarts(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._user_message_seen = True
    plugin._booted_mono = 0
    plugin._monotonic = lambda: 200
    plugin._wall = lambda: 1000000
    # Write 3 requested entries in the window
    plugin._state_dir.mkdir(parents=True, exist_ok=True)
    with open(plugin._history_path, "w") as f:
        for i in range(3):
            f.write(json.dumps({"event": "requested", "requested_at": 1000000 - i}) + "\n")
    refusal = plugin._loop_guard_refusal()
    assert refusal is not None
    assert "3 restarts" in refusal


async def test_loop_guard_allows_restart(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._user_message_seen = True
    plugin._booted_mono = 0
    plugin._monotonic = lambda: 200
    plugin._wall = lambda: 1000000
    refusal = plugin._loop_guard_refusal()
    assert refusal is None


# --- restart tool --------------------------------------------------------------


async def test_restart_refused_no_channel(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)

    ctx = MagicMock()
    ctx.channel = None
    result = await plugin._restart("test", ctx)
    assert "no channel" in result


async def test_restart_refused_loop_guard(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)

    ctx = MagicMock()
    ctx.channel = FakeChannel("test:home")
    result = await plugin._restart("test", ctx)
    assert "Refused" in result


async def test_restart_preflight_failure(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._user_message_seen = True
    plugin._booted_mono = 0
    plugin._monotonic = lambda: 200
    plugin._preflight = AsyncMock(return_value=(1, "config error: bad yaml"))

    ctx = MagicMock()
    ctx.channel = FakeChannel("test:home")
    result = await plugin._restart("test", ctx)
    assert "pre-flight failed" in result
    assert "config error" in result


# --- marker --------------------------------------------------------------------


async def test_marker_write_and_take(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._wall = lambda: 12345.0

    marker = {"reason": "test", "channel_id": "test:home"}
    plugin._write_marker(marker)
    assert plugin._marker_path.exists()

    taken = plugin._take_marker()
    assert taken is not None
    assert taken["reason"] == "test"
    assert not plugin._marker_path.exists()

    # History has a delivered entry
    records = read_jsonl(plugin._history_path)
    assert any(r.get("event") == "delivered" for r in records)


async def test_marker_missing(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    assert plugin._take_marker() is None


async def test_marker_corrupt(tmp_path, caplog):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._state_dir.mkdir(parents=True, exist_ok=True)
    plugin._marker_path.write_text("not json{")
    assert plugin._take_marker() is None
    assert not plugin._marker_path.exists()


# --- notification ---------------------------------------------------------------


async def test_notify_restart_delivers_to_channel(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._booted_wall = 1000.0
    plugin._wall = lambda: 1005.0

    marker = {
        "reason": "test restart",
        "channel_id": "test:home",
        "tool_call_id": "tc-123",
        "requested_at": 995.0,
    }
    await plugin._notify_restart(marker)

    assert len(pm.ahook.notifies) == 1
    n = pm.ahook.notifies[0]
    assert n["channel"].id == "test:home"
    assert n["tool_call_id"] == "tc-123"
    assert "Restart complete" in n["text"]
    assert "test restart" in n["text"]
    assert "5s" in n["text"]  # downtime


async def test_notify_restart_falls_back_to_home(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._booted_wall = 1000.0
    plugin._wall = lambda: 1005.0

    # Channel not in registry
    marker = {
        "reason": "test",
        "channel_id": "unknown:channel",
        "tool_call_id": "tc-456",
        "requested_at": 995.0,
    }
    await plugin._notify_restart(marker)

    assert len(pm.ahook.notifies) == 1
    n = pm.ahook.notifies[0]
    assert n["channel"].id == "test:home"  # fell back to home
    assert "no longer registered" in n["text"]


async def test_notify_home_no_channel_configured(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._home_channel_id = None

    await plugin._notify_home("test notice")
    assert len(pm.ahook.notifies) == 0


# --- safe mode -----------------------------------------------------------------


async def test_safe_mode_suffix(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._boot_mode = "safe-tier-1"
    plugin._blocked = ["plugin_a", "plugin_b"]

    suffix = plugin._safe_mode_suffix()
    assert "Safe mode" in suffix
    assert "tier 1" in suffix
    assert "plugin_a" in suffix


async def test_safe_mode_notice_reason(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._boot_mode = "safe-tier-1"

    reason = plugin._home_notice_reason()
    assert reason is not None
    assert "safe mode" in reason


async def test_abnormal_boot_notice_reason(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._boot_mode = "normal"
    plugin._previous_abnormal = True

    reason = plugin._home_notice_reason()
    assert reason is not None
    assert "abnormally" in reason


# --- config reload ---------------------------------------------------------------


async def test_config_reload_updates_settings(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    assert plugin._min_uptime == 120

    new_cfg = dict(cfg)
    new_cfg["restart"] = {**cfg["restart"], "min_uptime_seconds": 300}
    await plugin.on_config_reload(new_cfg)
    assert plugin._min_uptime == 300
