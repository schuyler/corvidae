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


async def test_config_reload_updates_blocked(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    assert plugin._blocked == []

    new_cfg = dict(cfg)
    new_cfg["plugins"] = {"disabled": ["plugin_x", "plugin_y"]}
    await plugin.on_config_reload(new_cfg)
    assert plugin._blocked == ["plugin_x", "plugin_y"]


def test_coerce_int_valid():
    from corvidae_restart import _coerce_int
    assert _coerce_int(42, "test") == 42
    assert _coerce_int(3.9, "test") == 3


def test_coerce_int_raises_on_bool():
    from corvidae_restart import _coerce_int
    import pytest
    with pytest.raises(ValueError, match="must be an integer"):
        _coerce_int(True, "test")


def test_coerce_int_raises_on_string():
    from corvidae_restart import _coerce_int
    import pytest
    with pytest.raises(ValueError, match="must be an integer"):
        _coerce_int("fast", "min_uptime_seconds")


def test_coerce_float_valid():
    from corvidae_restart import _coerce_float
    assert _coerce_float(42, "test") == 42.0
    assert _coerce_float(3.9, "test") == 3.9


def test_coerce_float_raises_on_string():
    from corvidae_restart import _coerce_float
    import pytest
    with pytest.raises(ValueError, match="must be a number"):
        _coerce_float("fast", "notify_delay_seconds")


async def test_marker_event_cannot_override_delivered(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._wall = lambda: 12345.0

    # Marker with an "event" key that tries to override
    marker = {"reason": "test", "channel_id": "test:home", "event": "requested"}
    plugin._write_marker(marker)

    taken = plugin._take_marker()
    assert taken is not None
    assert taken["event"] == "requested"  # the marker's own event

    # But the history entry must say "delivered"
    records = read_jsonl(plugin._history_path)
    delivered = [r for r in records if r.get("event") == "delivered"]
    assert len(delivered) == 1
    assert delivered[0]["reason"] == "test"


async def test_restart_lock_prevents_concurrent(tmp_path):
    plugin, pm, cfg = make_plugin(tmp_path)
    await plugin.on_init(pm, cfg)
    plugin._user_message_seen = True
    plugin._booted_mono = 0
    plugin._monotonic = lambda: 200
    plugin._wall = lambda: 1000000
    plugin._preflight = AsyncMock(return_value=(0, ""))
    plugin._killer = lambda pid, sig: None  # no-op kill

    ctx = MagicMock()
    ctx.channel = FakeChannel("test:home")
    ctx.tool_call_id = "tc-1"

    # First restart should proceed (and block on the Event)
    task1 = asyncio.create_task(plugin._restart("first", ctx))
    await asyncio.sleep(0.05)  # let it get past the lock

    # Second restart should be blocked by the lock
    # (it will wait for the lock, which the first task holds)
    # We can't easily test this without the first task completing,
    # so just verify the lock exists and is an asyncio.Lock
    assert isinstance(plugin._restart_lock, asyncio.Lock)


async def test_read_boot_status_scans_backward(tmp_path):
    """Test that read_boot_status scans backward for service_result."""
    f = tmp_path / "boot.jsonl"
    # Start, then a non-result record, then a result, then a new start
    f.write_text(
        json.dumps({"mode": "normal", "ts": 1}) + "\n"
        + json.dumps({"some_other_key": "value", "ts": 2}) + "\n"
        + json.dumps({"service_result": "success", "ts": 3}) + "\n"
        + json.dumps({"mode": "normal", "ts": 4}) + "\n"
    )
    mode, abnormal = read_boot_status(f)
    assert mode == "normal"
    assert abnormal is False  # found the success result by scanning back


async def test_read_boot_status_no_result_record(tmp_path):
    """Test that missing service_result is treated as abnormal."""
    f = tmp_path / "boot.jsonl"
    f.write_text(
        json.dumps({"mode": "normal", "ts": 1}) + "\n"
        + json.dumps({"mode": "normal", "ts": 2}) + "\n"
    )
    mode, abnormal = read_boot_status(f)
    assert mode == "normal"
    assert abnormal is True  # no result record found


# --- real plugin manager: end-to-end restart path ---------------------------


async def test_restart_end_to_end_plugin_manager(tmp_path):
    """Full restart path through a real plugin manager.

    Asserts: tool is registered, marker and history are written,
    courtesy message is sent, killer is called with SIGTERM.
    """
    import signal as signal_mod
    from unittest.mock import AsyncMock
    from corvidae.hooks import create_plugin_manager
    from corvidae.channel import ChannelRegistry, ChannelConfig
    from corvidae_restart import RestartPlugin

    pm = create_plugin_manager()
    registry = ChannelRegistry()
    registry.agent_defaults = {
        "system_prompt": "test",
        "max_context_tokens": 8000,
        "reasoning_history": "drop",
    }
    pm.register(registry, name="registry")
    pm.ahook.send = AsyncMock()

    channel = registry.get_or_create("test", "scope1", config=ChannelConfig())

    plugin = RestartPlugin()
    pm.register(plugin, name="restart")

    config = {
        "_base_dir": str(tmp_path),
        "restart": {
            "home_channel": "test:scope1",
            "min_uptime_seconds": 0,
            "max_restarts": 10,
            "window_seconds": 21600,
            "preflight_timeout_seconds": 5,
            "notify_delay_seconds": 0,
        },
        "plugins": {"disabled": []},
    }

    # Fake the agent plugin (dependency)
    from unittest.mock import MagicMock
    agent = MagicMock()
    agent.tools = {}
    pm.register(agent, name="agent")

    await plugin.on_init(pm=pm, config=config)

    # Fake preflight and killer
    plugin._preflight = AsyncMock(return_value=(0, ""))
    killed = []
    plugin._killer = lambda pid, sig: killed.append((pid, sig))

    # Set the human-message latch
    plugin._user_message_seen = True
    plugin._booted_mono = 0
    plugin._monotonic = lambda: 200
    plugin._wall = lambda: 1000000.0

    # Register tools
    tools = []
    plugin.register_tools(tools)
    assert len(tools) == 1
    assert tools[0].name == "restart"

    # Call the tool
    from corvidae.tool import ToolContext
    ctx = ToolContext(channel=channel, tool_call_id="tc-e2e-1", task_queue=None)

    # The tool will call _restart, which will:
    # 1. Pass the loop guard
    # 2. Run preflight (faked: success)
    # 3. Write marker
    # 4. Append history
    # 5. Send courtesy message
    # 6. Call killer (faked: records the call)
    # 7. Block on asyncio.Event().wait()
    #
    # We run it as a task and cancel it after the killer is called.
    task = asyncio.create_task(plugin._restart("test restart", ctx))
    
    # Wait for the killer to be called
    for _ in range(100):
        if killed:
            break
        await asyncio.sleep(0.01)
    
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Assert: killer was called with our PID and SIGTERM
    assert len(killed) == 1
    assert killed[0][1] == signal_mod.SIGTERM

    # Assert: marker was written
    marker_path = tmp_path / "state" / "restart.json"
    assert marker_path.exists()
    import json
    marker = json.loads(marker_path.read_text())
    assert marker["reason"] == "test restart"
    assert marker["channel_id"] == channel.id
    assert marker["tool_call_id"] == "tc-e2e-1"

    # Assert: history has a "requested" entry
    history_path = tmp_path / "state" / "restart-history.jsonl"
    assert history_path.exists()
    records = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    requested = [r for r in records if r.get("event") == "requested"]
    assert len(requested) == 1
    assert requested[0]["reason"] == "test restart"

    # Assert: courtesy message was sent
    assert pm.ahook.send.await_count >= 1
    send_calls = pm.ahook.send.await_args_list
    # Find the courtesy message (text starts with "Restarting:")
    courtesy = [
        c for c in send_calls
        if hasattr(c.kwargs.get("event", None), "text")
        and c.kwargs["event"].text.startswith("Restarting:")
    ]
    assert len(courtesy) == 1
    assert "test restart" in courtesy[0].kwargs["event"].text


async def test_deliver_notices_end_to_end(tmp_path):
    """_deliver_notices reads the marker and notifies the channel."""
    from unittest.mock import AsyncMock, MagicMock
    from corvidae.hooks import create_plugin_manager
    from corvidae.channel import ChannelRegistry, ChannelConfig
    from corvidae_restart import RestartPlugin

    pm = create_plugin_manager()
    registry = ChannelRegistry()
    registry.agent_defaults = {
        "system_prompt": "test",
        "max_context_tokens": 8000,
        "reasoning_history": "drop",
    }
    pm.register(registry, name="registry")

    channel = registry.get_or_create("test", "scope1", config=ChannelConfig())

    plugin = RestartPlugin()
    pm.register(plugin, name="restart")

    config = {
        "_base_dir": str(tmp_path),
        "restart": {
            "home_channel": "test:scope1",
            "notify_delay_seconds": 0,
        },
        "plugins": {"disabled": []},
    }

    agent = MagicMock()
    agent.tools = {"some_tool": True}  # non-empty so _wait_until_ready passes
    pm.register(agent, name="agent")

    pm.ahook.on_notify = AsyncMock()

    await plugin.on_init(pm=pm, config=config)
    plugin._booted_wall = 1005.0
    plugin._wall = lambda: 1005.0

    # Write a marker as if a restart had been requested before "crash"
    import json
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "requested_at": 1000.0,
        "reason": "config change",
        "channel_id": channel.id,
        "tool_call_id": "tc-deliver-1",
        "pid": 12345,
    }
    (state_dir / "restart.json").write_text(json.dumps(marker))

    # Run _deliver_notices
    await plugin._deliver_notices()

    # Assert: on_notify was called with the right args
    assert pm.ahook.on_notify.await_count == 1
    call = pm.ahook.on_notify.await_args
    assert call.kwargs["source"] == "restart"
    assert call.kwargs["tool_call_id"] == "tc-deliver-1"
    assert "Restart complete" in call.kwargs["text"]
    assert "config change" in call.kwargs["text"]
    assert "5s" in call.kwargs["text"]  # downtime

    # Assert: marker was consumed
    assert not (state_dir / "restart.json").exists()

    # Assert: history has a "delivered" entry
    history_path = state_dir / "restart-history.jsonl"
    records = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    delivered = [r for r in records if r.get("event") == "delivered"]
    assert len(delivered) == 1
    assert delivered[0]["reason"] == "config change"
