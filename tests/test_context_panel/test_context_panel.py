"""Server-side tests for the context panel plugin (issue #687).

Real ContextWindows and the kernel's own fit_to_budget, replace_with_summary
and apply_reasoning_history drive the change cases; requests go through a real
WebPlugin, so the panel's routes sit behind the console's authentication.
"""

from __future__ import annotations

import pytest
from context_panel import ContextPanelPlugin

from corvidae.channel import ChannelRegistry, ReasoningHistory
from corvidae.channels.web import WebPlugin
from corvidae.context import ContextWindow, MessageType
from corvidae.hooks import TransportPlugin, create_plugin_manager, install_plugin_hookspecs
from corvidae.outbound import OutboundProgress, OutboundThinking, OutboundToolStatus
from corvidae.thinking import apply_reasoning_history
from corvidae.tool_policy import ToolView
from tests.test_web_plugin import TOKEN, _no_env_token, make_env  # noqa: F401

PANEL = "/panel/context_panel"
ENABLED = {"context_panel": {}}


class EchoTransport(TransportPlugin):
    """A transport that is not web: owns testtx:* channels and shows nothing."""

    transport = "testtx"


async def panel_env(make_env, **kwargs):
    """A running web console with the context panel loaded and enabled."""
    plugin = ContextPanelPlugin()
    env = await make_env(
        extra_plugins={"context_panel": plugin, **kwargs.pop("extra_plugins", {})},
        extra_config=kwargs.pop("extra_config", ENABLED),
        **kwargs,
    )
    return env, plugin


def seed(env, rows=(), transport="web", scope="main", prompt="SYS") -> ContextWindow:
    """A loaded window on a registry channel, one MESSAGE row per dict."""
    channel = env.registry.get_or_create(transport, scope)
    win = ContextWindow(channel.id)
    win.system_prompt = prompt
    for row in rows:
        win.append(row)
    channel.conversation = win
    return win


def numbered(n: int, size: int = 0) -> list[dict]:
    """n distinct user rows; `size` pads each so a budget cut is possible."""
    return [{"role": "user", "content": f"row {i} " + "x" * size} for i in range(n)]


async def fetch(env, route: str, status: int = 200, **params):
    """GET a panel route with the console cookie; returns the JSON body."""
    async with env.http.get(f"{env.base}{PANEL}/{route}", params=params,
                            headers=env.cookie()) as r:
        assert r.status == status
        return await r.json() if status == 200 else None


async def open_window(env, cid: str, **params) -> dict:
    return await fetch(env, "window", channel=cid, **params)


async def poll(env, cid: str, cursor: dict | None = None) -> dict:
    params = {"channel": cid}
    if cursor is not None:
        params.update(count=cursor["count"], digest=cursor["digest"])
    return await fetch(env, "poll", **params)


async def status_of(env, path: str) -> int:
    async with env.http.get(f"{env.base}{path}", headers=env.cookie()) as r:
        return r.status


def build_pm(*plugins: tuple[str, object]):
    """A plugin manager with a registry and the given plugins, hookspecs installed."""
    pm = create_plugin_manager()
    registry = ChannelRegistry()
    pm.register(registry, name="registry")
    for name, plugin in plugins:
        pm.register(plugin, name=name)
    install_plugin_hookspecs(pm)
    return pm, registry


# ===========================================================================
# Config (R3, R4)
# ===========================================================================

class TestConfig:

    async def test_no_block_serves_no_route(self, make_env):
        env, _ = await panel_env(make_env, extra_config={})
        for path in ("/", "/channels", "/window", "/poll"):
            assert await status_of(env, f"{PANEL}{path}") == 404, path
        async with env.http.get(f"{env.base}/", headers=env.cookie()) as r:
            assert "context_panel" not in await r.text()

    async def test_no_block_buffers_no_live_events(self, make_env):
        env, plugin = await panel_env(make_env, extra_config={})
        seed(env)
        channel = env.registry.get("web:main")
        for event in (OutboundThinking(text="t"), OutboundProgress(text="p"),
                      OutboundToolStatus(tool_name="x", status="dispatched")):
            await env.pm.ahook.send(channel=channel, event=event)
        assert plugin._live == {}

    @pytest.mark.parametrize("with_web_block", [True, False])
    async def test_block_without_web_is_inert(self, with_web_block):
        plugin = ContextPanelPlugin()
        pm, registry = build_pm(("context_panel", plugin))
        config = {"context_panel": {}}
        if with_web_block:
            config["web"] = {"host": "127.0.0.1", "port": 0, "token": TOKEN}
        channel = registry.get_or_create("testtx", "room")
        channel.conversation = ContextWindow(channel.id)
        await pm.ahook.on_init(pm=pm, config=config)
        await pm.ahook.on_start(config=config)
        await pm.ahook.send(channel=channel, event=OutboundThinking(text="t"))
        await pm.ahook.on_stop()
        assert plugin._live == {}

    async def test_empty_block_enables(self, make_env):
        env, _ = await panel_env(make_env)
        async with env.http.get(f"{env.base}{PANEL}/", headers=env.cookie()) as r:
            assert r.status == 200
        async with env.http.get(f"{env.base}/", headers=env.cookie()) as r:
            assert f'<a href="{PANEL}/">' in await r.text()

    async def test_null_block_enables(self, make_env):
        env, _ = await panel_env(make_env, extra_config={"context_panel": None})
        async with env.http.get(f"{env.base}{PANEL}/", headers=env.cookie()) as r:
            assert r.status == 200
        async with env.http.get(f"{env.base}/", headers=env.cookie()) as r:
            assert f'<a href="{PANEL}/">' in await r.text()

    @pytest.mark.parametrize("value", [[1], "yes"])
    async def test_non_mapping_block_fails_naming_key(self, value):
        plugin = ContextPanelPlugin()
        pm, _ = build_pm(("context_panel", plugin))
        with pytest.raises(ValueError) as exc:
            await plugin.on_init(pm=pm, config={"context_panel": value})
        text = str(exc.value)
        assert "context_panel" in text
        assert repr(value) in text
        assert "check agent.yaml" in text

    @pytest.mark.parametrize("panel_first", [True, False])
    async def test_registered_under_real_broadcast_in_either_order(self, panel_first):
        plugin, web_plugin = ContextPanelPlugin(), WebPlugin()
        pair = [("context_panel", plugin), ("web", web_plugin)]
        pm, _ = build_pm(*(pair if panel_first else pair[::-1]))
        config = {"web": {"host": "127.0.0.1", "port": 0, "token": TOKEN},
                  "context_panel": {}}
        await pm.ahook.on_init(pm=pm, config=config)
        assert "context_panel" in [name for name, _ in web_plugin._panels]

    async def test_routes_are_three_gets(self, make_env):
        env, _ = await panel_env(make_env)
        panels = dict(env.plugin._panels)
        routes = panels["context_panel"].routes
        assert sorted((r.method, r.path) for r in routes) == [
            ("GET", "/channels"), ("GET", "/poll"), ("GET", "/window")]


# ===========================================================================
# Channel list and window (R5, R6, R10)
# ===========================================================================

class TestChannelsAndWindow:

    async def test_channel_list_reports_loaded_channels(self, make_env):
        env, _ = await panel_env(make_env)
        full = seed(env, numbered(3), scope="z")
        empty = seed(env, scope="a")
        env.registry.get_or_create("web", "m")  # registered, never loaded
        body = await fetch(env, "channels")
        assert [c["id"] for c in body["channels"]] == ["web:a", "web:z"]
        by_id = {c["id"]: c for c in body["channels"]}
        assert by_id["web:z"]["messages"] == len(full.messages) == 3
        assert by_id["web:a"]["messages"] == len(empty.messages) == 0
        for cid, win in (("web:z", full), ("web:a", empty)):
            assert by_id[cid]["tokens"] == win.token_estimate()
            channel = env.registry.get(cid)
            assert by_id[cid]["budget"] == env.registry.resolve_config(channel)["max_context_tokens"]

    async def test_window_returns_prompt_tools_and_typed_rows(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, prompt="the system prompt")
        schemas = [{"type": "function", "function": {"name": "lookup"}}]
        win.tool_view = ToolView(schemas=schemas)
        win.append({"role": "user", "content": "hello"})
        win.append({"role": "user", "content": "sum"}, MessageType.SUMMARY)
        win.append({"role": "system", "content": "ctx"}, MessageType.CONTEXT)
        win.append({"role": "assistant", "content": "withheld"}, MessageType.UNSENT)
        win.append({"role": "tool", "tool_call_id": "c1", "content": "result"})
        win.append({"role": "user", "content": "persisted"})
        win.messages[-1]["_db_id"] = 7
        body = await open_window(env, "web:main")
        assert body["loaded"] is True
        assert body["system_prompt"] == "the system prompt"
        assert body["tools"] == schemas
        assert [r["type"] for r in body["rows"]] == [
            "message", "summary", "context", "unsent", "message", "message"]
        assert [r["msg"]["content"] for r in body["rows"]] == [
            "hello", "sum", "ctx", "withheld", "result", "persisted"]
        assert body["rows"][4]["msg"] == {
            "role": "tool", "tool_call_id": "c1", "content": "result"}
        assert all(not k.startswith("_") for r in body["rows"] for k in r["msg"])
        assert body["total"] == 6
        assert body["first"] == 0
        assert body["cursor"]["count"] == 6
        assert body["cursor"]["digest"]

    async def test_window_unloaded_and_unregistered_channel(self, make_env):
        env, _ = await panel_env(make_env)
        env.registry.get_or_create("web", "pending")  # conversation is None
        assert await open_window(env, "web:pending") == {"loaded": False}
        assert await open_window(env, "discord:never-seen") == {"loaded": False}

    async def test_window_caps_to_newest_rows(self, make_env):
        env, _ = await panel_env(make_env)
        seed(env, numbered(250))
        capped = await open_window(env, "web:main")
        assert len(capped["rows"]) == 200
        assert capped["first"] == 50
        assert capped["total"] == 250
        assert capped["cursor"]["count"] == 250
        assert capped["rows"][0]["msg"]["content"].startswith("row 50 ")
        full = await open_window(env, "web:main", all="1")
        assert len(full["rows"]) == 250
        assert full["first"] == 0

    async def test_poll_never_carries_the_window(self, make_env):
        env, _ = await panel_env(make_env)
        seed(env, numbered(5))
        cursor = (await open_window(env, "web:main"))["cursor"]
        body = await poll(env, "web:main", cursor)
        assert body["rows"] == []
        assert set(body) <= {"loaded", "resync", "rows", "cursor", "live"}


# ===========================================================================
# Tail (R7)
# ===========================================================================

class TestTail:

    async def test_tail_append_returns_only_new_rows(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, numbered(3))
        cursor = (await open_window(env, "web:main"))["cursor"]
        win.append({"role": "system", "content": "injected"}, MessageType.CONTEXT)
        win.append({"role": "tool", "tool_call_id": "c1", "content": "out"})
        body = await poll(env, "web:main", cursor)
        assert body["resync"] is False
        assert [(r["type"], r["msg"]["content"]) for r in body["rows"]] == [
            ("context", "injected"), ("message", "out")]
        assert body["cursor"]["count"] == 5
        again = await poll(env, "web:main", body["cursor"])
        assert again["resync"] is False
        assert again["rows"] == []

    async def test_tail_db_id_assignment_does_not_resync(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, numbered(3))
        cursor = (await open_window(env, "web:main"))["cursor"]
        win.messages[-1]["_db_id"] = 42
        body = await poll(env, "web:main", cursor)
        assert body["resync"] is False
        assert body["rows"] == []

    async def test_tail_compaction_resyncs(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, numbered(6))
        cursor = (await open_window(env, "web:main"))["cursor"]
        win.replace_with_summary({"role": "user", "content": "summary"}, 2)
        body = await poll(env, "web:main", cursor)
        assert body["loaded"] is True
        assert body["resync"] is True

    async def test_tail_system_prompt_refresh_resyncs(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, numbered(6))
        win.tool_view = ToolView(schemas=[{"type": "function", "function": {"name": "a"}}])
        win.replace_with_summary({"role": "user", "content": "summary"}, 2)
        cursor = (await open_window(env, "web:main"))["cursor"]
        # on_compaction assigns the new prompt after the head has moved.
        win.system_prompt = "refreshed prompt"
        assert (await poll(env, "web:main", cursor))["resync"] is True
        cursor = (await open_window(env, "web:main"))["cursor"]
        assert (await poll(env, "web:main", cursor))["resync"] is False
        win.tool_view = ToolView(schemas=[{"type": "function", "function": {"name": "b"}}])
        assert (await poll(env, "web:main", cursor))["resync"] is True

    async def test_tail_head_trim_resyncs(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, numbered(10, size=200))
        cursor = (await open_window(env, "web:main"))["cursor"]
        dropped, truncated = win.fit_to_budget(win.token_estimate() // 2)
        assert dropped and not truncated
        assert (await poll(env, "web:main", cursor))["resync"] is True
        for row in numbered(len(dropped), size=200):
            win.append(row)
        assert len(win.messages) >= cursor["count"]
        assert (await poll(env, "web:main", cursor))["resync"] is True

    async def test_tail_newest_row_truncation_resyncs(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, [{"role": "user", "content": "word " * 2000}], prompt="")
        cursor = (await open_window(env, "web:main"))["cursor"]
        dropped, truncated = win.fit_to_budget(300)
        assert truncated and dropped == []
        assert (await poll(env, "web:main", cursor))["resync"] is True

    async def test_tail_interior_rewrite_resyncs(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "reasoning_content": "think 1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "reasoning_content": "think 2"},
        ])
        cursor = (await open_window(env, "web:main"))["cursor"]
        apply_reasoning_history(win.messages, ReasoningHistory.EXCHANGE)
        assert "reasoning_content" not in win.messages[1]
        assert "reasoning_content" in win.messages[3]
        assert (await poll(env, "web:main", cursor))["resync"] is True

    async def test_tail_unloaded_channel_reported(self, make_env):
        env, _ = await panel_env(make_env)
        channel = env.registry.get_or_create("web", "main")
        assert await poll(env, "web:main") == {"loaded": False}
        assert await poll(env, "web:unregistered") == {"loaded": False}
        channel.conversation = ContextWindow(channel.id)
        channel.conversation.append({"role": "user", "content": "first turn"})
        body = await poll(env, "web:main")
        assert body["loaded"] is True
        assert body["resync"] is True


# ===========================================================================
# Live events (R8)
# ===========================================================================

class TestLive:

    async def test_live_events_from_non_web_transport_reach_poll(self, make_env):
        env, _ = await panel_env(make_env, extra_plugins={"testtx": EchoTransport()})
        seed(env, numbered(2), transport="testtx", scope="room")
        channel = env.registry.get("testtx:room")
        for event in (
            OutboundThinking(text="reasoning text"),
            OutboundProgress(text="progress text"),
            OutboundToolStatus(tool_name="web_search", status="dispatched",
                               args_summary="q=cats", tool_call_id="call_1"),
        ):
            await env.pm.ahook.send(channel=channel, event=event)
        body = await poll(env, "testtx:room")
        live = body["live"]
        assert [e["kind"] for e in live] == ["thinking", "progress", "tool_status"]
        assert live[0]["text"] == "reasoning text"
        assert live[1]["text"] == "progress text"
        assert live[2]["tool"] == "web_search"
        assert live[2]["status"] == "dispatched"
        assert live[2]["summary"] == "q=cats"
        assert live[2]["tool_call_id"] == "call_1"
        seqs = [e["seq"] for e in live]
        assert seqs == sorted(seqs) and len(set(seqs)) == 3

    async def test_live_events_clear_when_rows_land(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, numbered(3))
        channel = env.registry.get("web:main")
        await env.pm.ahook.send(channel=channel, event=OutboundThinking(text="first"))
        await env.pm.ahook.send(channel=channel, event=OutboundProgress(text="second"))
        assert len((await poll(env, "web:main"))["live"]) == 2
        win.append({"role": "tool", "tool_call_id": "c1", "content": "out"})
        assert (await poll(env, "web:main"))["live"] == []
        await env.pm.ahook.send(channel=channel, event=OutboundProgress(text="later"))
        live = (await poll(env, "web:main"))["live"]
        assert [e["text"] for e in live] == ["later"]

    async def test_live_events_survive_in_place_strip(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, numbered(2))
        win.append({"role": "assistant", "content": "reply", "reasoning_content": "secret thought"})
        channel = env.registry.get("web:main")
        await env.pm.ahook.send(channel=channel, event=OutboundThinking(text="secret thought"))
        apply_reasoning_history(win.messages, ReasoningHistory.DROP)
        assert "reasoning_content" not in win.messages[-1]
        live = (await poll(env, "web:main"))["live"]
        assert [e["text"] for e in live] == ["secret thought"]

    async def test_live_events_clear_on_compaction_at_same_length(self, make_env):
        env, _ = await panel_env(make_env)
        win = seed(env, numbered(4))
        channel = env.registry.get("web:main")
        await env.pm.ahook.send(channel=channel, event=OutboundThinking(text="old turn"))
        assert len((await poll(env, "web:main"))["live"]) == 1
        win.replace_with_summary({"role": "user", "content": "summary"}, 1)
        win.append({"role": "user", "content": "a"})
        win.append({"role": "user", "content": "b"})
        assert len(win.messages) == 4
        assert (await poll(env, "web:main"))["live"] == []



async def log_row(env, channel_id: str, ts: float) -> int:
    cursor = await env.persistence.db.execute(
        "INSERT INTO message_log (channel_id, message, timestamp, message_type) "
        "VALUES (?, ?, ?, 'message')",
        (channel_id, '{"role": "user", "content": "x"}', ts),
    )
    await env.persistence.db.commit()
    return cursor.lastrowid


async def test_rows_carry_log_timestamp(make_env):
    env, _ = await panel_env(make_env)
    win = seed(env, numbered(2))
    win.messages[0]["_db_id"] = await log_row(env, "web:main", 1700000000.5)
    body = await open_window(env, "web:main")
    assert body["rows"][0]["ts"] == 1700000000.5
    assert "ts" not in body["rows"][1]
    win.append({"role": "user", "content": "new"})
    win.messages[-1]["_db_id"] = await log_row(env, "web:main", 1700000200.0)
    polled = await poll(env, "web:main", body["cursor"])
    assert [r["ts"] for r in polled["rows"]] == [1700000200.0]
